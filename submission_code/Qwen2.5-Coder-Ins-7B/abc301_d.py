# YOUR CODE HERE
import sys
input = sys.stdin.read
data = input().split()
S = data[0]
N = int(data[1])

def max_value_less_than_or_equal_to_N(S, N):
    def binary_value(s):
        return int(s, 2)
    
    def replace_question_marks(s, i, value):
        if i == len(s):
            return s
        if s[i] == '?':
            return replace_question_marks(s, i + 1, value) + str(value)
        else:
            return s[i] + replace_question_marks(s, i + 1, value)
    
    max_value = -1
    for i in range(2):
        for j in range(2):
            for k in range(2):
                new_S = replace_question_marks(S, 0, i)
                new_S = replace_question_marks(new_S, 1, j)
                new_S = replace_question_marks(new_S, 2, k)
                value = binary_value(new_S)
                if value <= N and value > max_value:
                    max_value = value
    return max_value

result = max_value_less_than_or_equal_to_N(S, N)
print(result)