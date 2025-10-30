# YOUR CODE HERE
import sys

def count_consecutive(s, l, r):
    count = 0
    for i in range(l, r):
        if s[i] == s[i + 1]:
            count += 1
    return count

def main():
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    Q = int(data[1])
    S = data[2]
    
    queries = []
    for i in range(Q):
        l = int(data[3 + 2 * i]) - 1
        r = int(data[4 + 2 * i]) - 1
        queries.append((l, r))
    
    for l, r in queries:
        print(count_consecutive(S, l, r))

if __name__ == "__main__":
    main()