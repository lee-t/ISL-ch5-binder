import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import pandas as pd

# Set random seed for reproducibility
np.random.seed(42)

# Generate synthetic data
n_cells_A = 50
n_cells_B = 45

# Cell type A: lower expression (mean=2.5)
expression_A = np.random.gamma(shape=2, scale=1.25, size=n_cells_A)

# Cell type B: higher expression (mean=4.0)  
expression_B = np.random.gamma(shape=3, scale=1.33, size=n_cells_B)

# Combine data
all_expression = np.concatenate([expression_A, expression_B])
cell_types = ['Cell Type A'] * n_cells_A + ['Cell Type B'] * n_cells_B

# Calculate observed difference in means
observed_diff = np.mean(expression_B) - np.mean(expression_A)

# Permutation test
n_permutations = 10000
permuted_diffs = []

for i in range(n_permutations):
    # Shuffle the cell type labels
    shuffled_labels = np.random.permutation(cell_types)
    
    # Calculate difference in means for shuffled data
    group_A_shuffled = all_expression[np.array(shuffled_labels) == 'Cell Type A']
    group_B_shuffled = all_expression[np.array(shuffled_labels) == 'Cell Type B']
    
    diff_shuffled = np.mean(group_B_shuffled) - np.mean(group_A_shuffled)
    permuted_diffs.append(diff_shuffled)

permuted_diffs = np.array(permuted_diffs)

# Calculate p-value
p_value = np.sum(np.abs(permuted_diffs) >= np.abs(observed_diff)) / n_permutations

# Create the figure
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Permutation Test for Differential Gene Expression\nGene X between Cell Type A and Cell Type B', 
             fontsize=16, fontweight='bold')

# Panel 1: Original data distribution
ax1 = axes[0, 0]
data_df = pd.DataFrame({
    'Expression': all_expression,
    'Cell_Type': cell_types
})

sns.violinplot(data=data_df, x='Cell_Type', y='Expression', ax=ax1, palette=['lightcoral', 'skyblue'])
sns.stripplot(data=data_df, x='Cell_Type', y='Expression', ax=ax1, 
              color='black', alpha=0.6, size=3)

ax1.set_title('A) Original Data: Gene X Expression', fontweight='bold')
ax1.set_ylabel('Expression Level')
ax1.axhline(np.mean(expression_A), color='red', linestyle='--', alpha=0.7, 
            label=f'Mean A = {np.mean(expression_A):.2f}')
ax1.axhline(np.mean(expression_B), color='blue', linestyle='--', alpha=0.7, 
            label=f'Mean B = {np.mean(expression_B):.2f}')
ax1.legend()

# Panel 2: Permutation process illustration
ax2 = axes[0, 1]
# Show a few examples of permuted data
n_examples = 5
example_diffs = permuted_diffs[:n_examples]

bars = ax2.bar(range(n_examples), example_diffs, color='lightgray', alpha=0.7)
ax2.axhline(observed_diff, color='red', linewidth=3, 
            label=f'Observed difference = {observed_diff:.2f}')
ax2.set_title('B) Examples of Permuted Differences', fontweight='bold')
ax2.set_xlabel('Permutation Example')
ax2.set_ylabel('Difference in Means (B - A)')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Panel 3: Null distribution from permutations
ax3 = axes[1, 0]
ax3.hist(permuted_diffs, bins=50, alpha=0.7, color='lightgray', density=True, 
         label='Null Distribution')
ax3.axvline(observed_diff, color='red', linewidth=3, 
            label=f'Observed difference = {observed_diff:.2f}')
ax3.axvline(-observed_diff, color='red', linewidth=3, linestyle='--', alpha=0.7)

# Shade the rejection region
extreme_values = permuted_diffs[np.abs(permuted_diffs) >= np.abs(observed_diff)]
if len(extreme_values) > 0:
    ax3.hist(extreme_values, bins=20, alpha=0.8, color='red', density=True, 
             label=f'p-value region')

ax3.set_title('C) Null Distribution from Permutations', fontweight='bold')
ax3.set_xlabel('Difference in Means (B - A)')
ax3.set_ylabel('Density')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Panel 4: Summary statistics
ax4 = axes[1, 1]
ax4.axis('off')

summary_text = f"""
Statistical Summary:

• Sample sizes: n_A = {n_cells_A}, n_B = {n_cells_B}

• Observed means:
  Cell Type A: {np.mean(expression_A):.3f}
  Cell Type B: {np.mean(expression_B):.3f}

• Observed difference: {observed_diff:.3f}

• Permutations: {n_permutations:,}

• P-value: {p_value:.4f}

• Interpretation: 
  {'Reject H₀' if p_value < 0.05 else 'Fail to reject H₀'} 
  (α = 0.05)
"""

ax4.text(0, 0.9, summary_text, transform=ax4.transAxes, fontsize=12,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.3))

plt.tight_layout()
plt.show()

print(f"Permutation test results:")
print(f"Observed difference in means: {observed_diff:.4f}")
print(f"P-value: {p_value:.4f}")
print(f"Conclusion: {'Statistically significant' if p_value < 0.05 else 'Not statistically significant'} at α = 0.05")
