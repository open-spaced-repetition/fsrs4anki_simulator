import gradio as gr
from gradio.components import Textbox, Slider, Plot
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

plt.style.use('seaborn-v0_8-whitegrid')

columns = ["difficulty", "stability", "retrievability", "delta_t",
           "reps", "lapses", "last_date", "due", "ivl", "cost", "rand"]
col = {key: i for i, key in enumerate(columns)}


def simulate(w, request_retention=0.9, deck_size=10000, learn_span=100, max_cost_perday=200, max_ivl=36500, recall_cost=10, forget_cost=30, learn_cost=10):
    card_table = np.zeros((len(columns), deck_size))
    card_table[col["due"]] = learn_span
    card_table[col["difficulty"]] = 1e-10
    card_table[col["stability"]] = 1e-10

    review_cnt_per_day = np.zeros(learn_span)
    learn_cnt_per_day = np.zeros(learn_span)
    memorized_cnt_per_day = np.zeros(learn_span)

    def cal_next_recall_stability(s, r, d, response):
        if response == 1:
            return s * (1 + np.exp(w[8]) * (11 - d) * np.power(s, -w[9]) * (np.exp((1 - r) * w[10]) - 1))
        else:
            return np.maximum(0.1, np.minimum(w[11] * np.power(d, -w[12]) * (np.power(s + 1, w[13]) - 1) * np.exp((1 - r) * w[14]), s))

    for today in tqdm(range(learn_span)):
        has_learned = card_table[col["stability"]] > 1e-10
        card_table[col["delta_t"]][has_learned] = today - \
            card_table[col["last_date"]][has_learned]
        card_table[col["retrievability"]][has_learned] = np.power(
            1 + card_table[col["delta_t"]][has_learned] / (9 * card_table[col["stability"]][has_learned]), -1)

        card_table[col["cost"]] = 0
        need_review = card_table[col["due"]] <= today
        card_table[col["rand"]][need_review] = np.random.rand(
            np.sum(need_review))
        forget = card_table[col["rand"]] > card_table[col["retrievability"]]
        card_table[col["cost"]][need_review & forget] = forget_cost
        card_table[col["cost"]][need_review & ~forget] = recall_cost
        true_review = need_review & (
            np.cumsum(card_table[col["cost"]]) <= max_cost_perday)
        card_table[col["last_date"]][true_review] = today

        card_table[col["lapses"]][true_review & forget] += 1
        card_table[col["reps"]][true_review & ~forget] += 1

        card_table[col["stability"]][true_review & forget] = cal_next_recall_stability(
            card_table[col["stability"]][true_review & forget], card_table[col["retrievability"]][true_review & forget], card_table[col["difficulty"]][true_review & forget], 0)

        card_table[col["stability"]][true_review & ~forget] = cal_next_recall_stability(
            card_table[col["stability"]][true_review & ~forget], card_table[col["retrievability"]][true_review & ~forget], card_table[col["difficulty"]][true_review & ~forget], 1)

        card_table[col["difficulty"]][true_review & forget] = np.clip(
            card_table[col["difficulty"]][true_review & forget] + 2 * w[6], 1, 10)

        need_learn = card_table[col["due"]] == learn_span
        card_table[col["cost"]][need_learn] = learn_cost
        true_learn = need_learn & (
            np.cumsum(card_table[col["cost"]]) <= max_cost_perday)
        card_table[col["last_date"]][true_learn] = today
        first_ratings = np.random.randint(0, 4, np.sum(true_learn))
        card_table[col["stability"]][true_learn] = np.choose(
            first_ratings, w[:4])
        card_table[col["difficulty"]][true_learn] = w[4] - \
            w[5] * (first_ratings - 3)

        card_table[col["ivl"]][true_review | true_learn] = np.clip(np.round(
            9 * card_table[col["stability"]][true_review | true_learn] * (1 / request_retention - 1), 0), 1, max_ivl)
        card_table[col["due"]][true_review | true_learn] = today + \
            card_table[col["ivl"]][true_review | true_learn]

        review_cnt_per_day[today] = np.sum(true_review)
        learn_cnt_per_day[today] = np.sum(true_learn)
        memorized_cnt_per_day[today] = card_table[col["retrievability"]].sum()
    return card_table, review_cnt_per_day, learn_cnt_per_day, memorized_cnt_per_day


def interface_func(weights: str, learning_time: int, learn_span: int, deck_size: int, max_ivl: int, recall_cost: int, forget_cost: int, learn_cost: int,
                   progress=gr.Progress(track_tqdm=True)):
    plt.close('all')
    np.random.seed(42)
    weights = weights.replace('[', '').replace(']', '')
    w = list(map(lambda x: float(x.strip()), weights.split(',')))
    max_cost_perday = learning_time * 60

    def moving_average(data, window_size=learn_span//20):
        weights = np.ones(window_size) / window_size
        return np.convolve(data, weights, mode='valid')

    for request_retention in [0.95, 0.9, 0.85, 0.8, 0.75]:
        (_,
         review_cnt_per_day,
         learn_cnt_per_day,
         memorized_cnt_per_day) = simulate(w,
                                           request_retention=request_retention,
                                           deck_size=deck_size,
                                           learn_span=learn_span,
                                           max_cost_perday=max_cost_perday,
                                           max_ivl=max_ivl,
                                           recall_cost=recall_cost,
                                           forget_cost=forget_cost,
                                           learn_cost=learn_cost)

        plt.figure(1)
        plt.plot(moving_average(review_cnt_per_day),
                 label=f"R={request_retention*100:.0f}%")
        plt.title("Review Count per Day")
        plt.legend()
        plt.figure(2)
        plt.plot(moving_average(learn_cnt_per_day),
                 label=f"R={request_retention*100:.0f}%")
        plt.title("Learn Count per Day")
        plt.legend()
        plt.figure(3)
        plt.plot(np.cumsum(learn_cnt_per_day),
                 label=f"R={request_retention*100:.0f}%")
        plt.title("Cumulative Learn Count")
        plt.legend()
        plt.figure(4)
        plt.plot(memorized_cnt_per_day,
                 label=f"R={request_retention*100:.0f}%")
        plt.title("Memorized Count per Day")
        plt.legend()

    return plt.figure(1), plt.figure(2), plt.figure(3), plt.figure(4)

description = f"""
# FSRS4Anki Simulator

Here is a simulator for FSRS4Anki. It can simulate the learning process of a deck with given weights and parameters. 

It will help you to find the expected requestRetention for FSRS4Anki.

The simulator assumes that you spend the same amount of time on Anki every day.
"""

with gr.Blocks() as demo:
    with gr.Box():
        gr.Markdown(description)
    with gr.Box():
        with gr.Row():
            with gr.Column():
                weights = Textbox(label="Weights", lines=1,
                                  value="0.4, 0.6, 2.4, 5.8, 4.93, 0.94, 0.86, 0.01, 1.49, 0.14, 0.94, 2.18, 0.05, 0.34, 1.26, 0.29, 2.61")
                learning_time = Slider(label="Learning Time perday (minutes)",
                                       minimum=5, maximum=1440, step=5, value=30)
                learn_span = Slider(label="Learning Period (days)", minimum=30,
                                    maximum=3650, step=10, value=365)
                deck_size = Slider(label="Deck Size (cards)", minimum=100,
                                   maximum=100000, step=100, value=10000)
            with gr.Column():
                max_ivl = Slider(label="Maximum Interval (days)", minimum=30,
                                 maximum=36500, step=10, value=36500)
                recall_cost = Slider(label="Review Cost (seconds)", minimum=1,
                                     maximum=600, step=1, value=10)
                forget_cost = Slider(label="Relearn Cost (seconds)",
                                     minimum=1, maximum=600, step=1, value=30)
                learn_cost = Slider(label="Learn Cost (seconds)", minimum=1,
                                    maximum=600, step=1, value=10)
        with gr.Row():
            btn_plot = gr.Button("Simulate")
        with gr.Row():
            with gr.Column():
                review_count = Plot(label="Review Count per Day")
                learn_count = Plot(label="Learn Count per Day")
            with gr.Column():
                cumulative_learn_count = Plot(label="Cumulative Learn Count")
                memorized_count = Plot(label="Memorized Count per Day")

    btn_plot.click(
        fn=interface_func,
        inputs=[weights,
                learning_time,
                learn_span,
                deck_size,
                max_ivl,
                recall_cost,
                forget_cost,
                learn_cost,
                ],
        outputs=[review_count, learn_count,
                cumulative_learn_count, memorized_count],
    )

demo.queue().launch(show_error=True)
