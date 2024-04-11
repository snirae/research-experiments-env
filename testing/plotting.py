import matplotlib.pyplot as plt


def plot_runs_comparison(data_list, lookback_size, intervals=(0.5, 0.9), save_dir=None, name=None):
    num_elements = data_list[0]['label'].shape[1]

    num_cols = 1
    num_rows = (num_elements + num_cols - 1) // num_cols

    fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols,
                             figsize=(25, 10 * num_rows))

    for element in range(num_elements):
        row = element // num_cols
        col = element % num_cols
        # ax = axes[row, col] if num_rows > 1 else axes[col]
        ax = axes[row]

        for i, data in enumerate(data_list):
            label_data = data['label'][:, element]
            forecast_data = data['forecast'][:, element]
            plot_next_multi(ax, label_data, forecast_data, lookback_size, i, intervals=intervals, show_label=(i == 0))

        ax.set_title(f"Element {element + 1}")

    for element in range(num_elements, num_rows * num_cols):
        row = element // num_cols
        col = element % num_cols
        fig.delaxes(axes[row, col] if num_rows > 1 else axes[col])

    plt.tight_layout()
    
    # Save the plot
    if save_dir is not None and name is not None:
        plt.savefig(f"{save_dir}/{name}.png")
    else:
        plt.show()


def plot_next_multi(ax, label_data, forecast_data, lookback_size, run_index, intervals=None, show_label=True):
    if show_label:
        ax.plot(label_data, label='Target', color='brown')
    ax.plot(forecast_data, label='Prediction', color='blue')

    # plot lookback window size
    ax.axvline(lookback_size, color='red', linestyle='--', label='Lookback Window')

    if intervals is not None:
        lower = forecast_data - intervals[0]
        upper = forecast_data + intervals[1]
        ax.fill_between(range(len(forecast_data)), lower, upper, alpha=0.6, color='lightblue',
                        label=f"Intervals ({intervals[0]}, {intervals[1]})")

    ax.set_xlim(0, len(forecast_data))
    ax.legend()
    ax.set_xlabel('Time')
    ax.set_ylabel('Value')
    ax.set_title(f"Forecast - Run {run_index + 1}")
