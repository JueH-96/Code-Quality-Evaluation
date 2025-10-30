import sys
from collections import defaultdict

def main():
    input = sys.stdin.read
    data = input().split()
    idx = 0
    N = int(data[idx])
    T = int(data[idx+1])
    idx += 2
    A = []
    B = []
    for _ in range(T):
        A.append(int(data[idx]))
        B.append(int(data[idx+1]))
        idx += 2
    
    scores = defaultdict(int)
    count = defaultdict(int)
    unique = 0
    
    for i in range(T):
        a = A[i]
        b = B[i]
        prev_score = scores[a]
        if prev_score != 0:
            count[prev_score] -= 1
            if count[prev_score] == 0:
                unique -= 1
        new_score = prev_score + b
        scores[a] = new_score
        if new_score != 0:
            if count[new_score] == 0:
                unique += 1
            count[new_score] += 1
        print(unique)

if __name__ == "__main__":
    main()