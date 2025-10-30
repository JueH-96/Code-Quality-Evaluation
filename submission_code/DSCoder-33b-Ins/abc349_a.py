# YOUR CODE HERE
N = int(input())
A = list(map(int, input().split()))

# The final score of person N is the sum of all the scores
# because each game increases the winner's score by 1 and decreases the loser's score by 1
# so the sum of all the scores is the final score of person N
print(sum(A))