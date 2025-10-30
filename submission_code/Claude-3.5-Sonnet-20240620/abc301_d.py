# YOUR CODE HERE
def replace_question_marks(s, n):
    result = 0
    for i, char in enumerate(reversed(s)):
        if char == '1':
            result |= 1 << i
        elif char == '?':
            if result | (1 << i) <= n:
                result |= 1 << i
    return result if result <= n else -1

s = input().strip()
n = int(input())

print(replace_question_marks(s, n))