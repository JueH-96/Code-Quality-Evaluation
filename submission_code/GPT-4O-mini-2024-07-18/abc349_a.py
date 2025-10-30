# YOUR CODE HERE
def final_score_of_person_n(N, A):
    total_score = sum(A)
    final_score_n = -total_score
    return final_score_n

import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
A = list(map(int, data[1:N]))

result = final_score_of_person_n(N, A)
print(result)