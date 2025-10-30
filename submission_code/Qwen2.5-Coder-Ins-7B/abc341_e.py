# YOUR CODE HERE
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
Q = int(data[1])
S = data[2]
queries = data[3:]

def is_good_string(s):
    for i in range(len(s) - 1):
        if s[i] == s[i + 1]:
            return False
    return True

def flip_string(s, L, R):
    return ''.join('0' if c == '1' else '1' for c in s[L-1:R])

def extract_string(s, L, R):
    return s[L-1:R]

results = []

for query in queries:
    q_type = int(query[0])
    L = int(query[1])
    R = int(query[2])
    
    if q_type == 1:
        S = flip_string(S, L, R)
    elif q_type == 2:
        S_prime = extract_string(S, L, R)
        if is_good_string(S_prime):
            results.append("Yes")
        else:
            results.append("No")

for result in results:
    print(result)