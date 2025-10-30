# Read the input
a_scores = [int(x) for x in input().split()]
b_scores = [int(x) for x in input().split()]

# Calculate the total scores for each team
a_total = sum(a_scores)
b_total = sum(b_scores)

# If Team Aoki is already behind, they need to score at least 1 more run than Team Takahashi
if b_total < a_total:
    print(a_total - b_total + 1)
else:
    print(0)