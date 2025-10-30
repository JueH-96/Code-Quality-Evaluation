# YOUR CODE HERE
def check_substring(W, B):
    pattern = "wbwbwwbwbwbw"
    for i in range(len(pattern)):
        for j in range(i, len(pattern)):
            substring = pattern[i:j+1]
            if substring.count('w') == W and substring.count('b') == B:
                return "Yes"
    return "No"

W, B = map(int, input().split())
print(check_substring(W, B))