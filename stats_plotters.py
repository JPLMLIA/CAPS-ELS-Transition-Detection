"""
Utility functions for plotting for the evaluation framework.

Author: Ameya Daigavane
"""

from __future__ import division
import numpy as np

class StatsPlotter(object):
    """
    Handles plotting for the evaluation framework.
    It inherits all member variables from the Metric class.
    """

    def __init__(self, MetricObj=None):
        if MetricObj is not None:
            self.__dict__.update(vars(MetricObj))


    # Selector function for different plot types.
    def plot(self, fig, ax, plot_type, title_params={}, legend=None):
        if plot_type == 'roc':
            self.plot_roc(fig, ax, title_params, legend)
        elif plot_type == 'pr':
            self.plot_pr(fig, ax, title_params, legend)
        elif plot_type == 'td':
            self.plot_tdiffs(fig, ax, title_params, legend)
        else:
            raise ValueError('Invalid choice %s for parameter plot_type.' % plot_type)


    # Plot Receiver Operating Characteristic Curve.
    def plot_roc(self, fig, ax, title_params={}, color='tab:blue', legend=None, legend_title=None):
        ax.plot(self.fpr_list, self.tpr_list, '-o', color=color, label=legend)
        ax.plot([0, 1], [0, 1], 'gray', dashes=[6, 4])
        ax.set_title('%s ROC Curve \n Computed AUROC = %0.3f' %
                     (title_params.get('algorithm_name', ''), self.computed_auroc))
        ax.set_xlabel('False Positive Rate (FPR)')
        ax.set_xlim(0, 1.005)
        ax.set_ylabel('True Positive Rate (TPR)')
        ax.set_ylim(0, 1.005)
        ax.legend(title=legend_title)
        ax.set_aspect('equal', adjustable='box')


    # Plot Precision-Recall Curve.
    def plot_pr(self, fig, ax, title_params={}, color='tab:red', legend=None, legend_title=None):
        ax.plot(self.recall_list, self.precision_list, '-o', label=legend)
        ax.set_xlabel('Recall')
        ax.set_xlim(0, 1.005)
        ax.set_ylabel('Precision')
        ax.set_ylim(0, 1.005)
        ax.set_aspect('equal', adjustable='box')


    # Plot Time-Differences Curve.
    def plot_tdiffs(self, fig, ax, title_params={}, legend=None):

        for change_point_type in ['labelled']:
            time_differences_list = self.time_differences[change_point_type]

            mins = [np.percentile(diffs, 0)   for diffs in time_differences_list]
            q1   = [np.percentile(diffs, 25)  for diffs in time_differences_list]
            q2   = [np.percentile(diffs, 50)  for diffs in time_differences_list]
            q3   = [np.percentile(diffs, 75)  for diffs in time_differences_list]
            maxs = [np.percentile(diffs, 100) for diffs in time_differences_list]

            # Plot time differences.
            plot_percentiles = np.linspace(0, 100, len(self.thresholds))
            ax.plot(plot_percentiles, q2, 'k', linewidth=3)
            ax.fill_between(plot_percentiles, mins, q1, facecolor='gray', alpha=0.2)
            ax.fill_between(plot_percentiles, q1, q3, facecolor='gray', alpha=0.3)
            ax.fill_between(plot_percentiles, q1, maxs, facecolor='gray', alpha=0.2)

            ax.margins(x=0, y=0.005)
            ax.set_xticks(plot_percentiles[::20].astype(int))
            ax.set_xticklabels(plot_percentiles[::20].astype(int))
            ax.set_xlabel('Thresholds (Percentile)')
            ax.set_ylabel('Time (Seconds)')
            ax.set_aspect('auto', adjustable='datalim')

            # Plot false positives, predicted change-points that were not the closest to any labelled change-point.
            fp_ax = ax.twinx()
            color = 'tab:red'
            fp_ax.margins(x=0, y=0.005)
            fp_ax.set_ylabel('False Positives Rate', color=color)
            fp_ax.plot(plot_percentiles, self.time_differences['false_positives_rate'], color=color)
            fp_ax.tick_params(axis='y', labelcolor=color)

            ax.set_title('%s Time Differences for %s Change-Points' %
                         (title_params.get('algorithm_name', ''), change_point_type.capitalize()))


    # Plot scores.
    def plot_scores(self, fig, ax, times, scores, title_params=None,
                    color='tab:blue', legend=None, legend_title=None):
        ax.plot(times, scores, label=legend, color=color)
        ax.set_ylabel('Scores Assigned')
        ax.xaxis.set_tick_params(labelsize=8)
        ax.margins(x=0)
        if legend is not None:
            ax.legend(title=legend_title)


    # Plot intervals.
    def plot_intervals(self, fig, ax, times, scores, title_params=None):

        # Consider the 'middle' threshold.
        index = len(self.thresholds) // 2
        threshold = self.thresholds[index]
        positive_intervals = self.positive_intervals_list[index]
        predicted_positive_intervals = self.predicted_positive_intervals_list[index]
        true_positive_rate = self.tpr_list[index]
        false_positive_rate = self.fpr_list[index]

        # Regions around actual change-points in red.
        for interval in positive_intervals:
            ax.axvspan(xmin=interval[0], xmax=interval[1], color='r', alpha=0.2)

        # Regions around predicted change-points in blue.
        for interval in predicted_positive_intervals:
            ax.axvspan(xmin=interval[0], xmax=interval[1], color='b', alpha=0.2)

        # Assigned scores on top of these intervals.
        ax.plot(times, scores)
        ax.set_ylabel('Scores Assigned')
        ax.xaxis.set_tick_params(labelsize=8)
        ax.margins(x=0)

        # Set title.
        if title_params is not None:
            fig.text(s=' %s Evaluation \n %s \n Error Window = %d seconds \n Threshold = %0.2f \n TPR = %0.2f, FPR = %0.2f' %
                     (title_params['algorithm_name'],
                     title_params['parameters'],
                     title_params['error_window_seconds'],
                     threshold, true_positive_rate, false_positive_rate),
                     x=0.5, y=0.05, fontsize=13, horizontalalignment='center')
        fig.subplots_adjust(bottom=0.4, left=0.15)


    # Plot labelled anomalies.
    def plot_labels(self, fig, ax, labelled_anomalies, interval_width_minutes):
        interval_width_days = interval_width_minutes / 1440
        ax.set_ylabel('Labelled Crossings')
        for labelled_anomaly in labelled_anomalies:
            ax.axvspan(labelled_anomaly - interval_width_days/2,
                       labelled_anomaly + interval_width_days/2,
                       facecolor='g', alpha=0.5)
