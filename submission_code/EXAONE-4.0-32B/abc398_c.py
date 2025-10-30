from collections import Counter
import sys

def main():
    data = sys.stdin.read().split()
    n = int(data[0])
    A = list(map(int, data[1:1+n]))
    
    freq = Counter(A)
    
    max_val = -1
    ans = -1
    for i in range(n):
        if freq[A[i]] == 1:
            if A[i] > max_val:
                max_val = A[i]
                ans = i + 1
                
    print(ans if ans != -1 else -1)

if __name__ == "__main__":
    main()