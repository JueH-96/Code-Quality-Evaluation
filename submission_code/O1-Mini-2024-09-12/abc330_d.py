# YOUR CODE HERE
import sys

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    S = data[1:]
    
    row_counts = [s.count('o') for s in S]
    col_counts = [0] * N
    for s in S:
        for j, c in enumerate(s):
            if c == 'o':
                col_counts[j] +=1
    total = 0
    for i in range(N):
        s = S[i]
        for j in range(N):
            if s[j] == 'o':
                total += (row_counts[i]-1) * (col_counts[j]-1)
    print(total)

if __name__ == "__main__":
    main()