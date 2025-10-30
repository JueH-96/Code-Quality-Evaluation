# YOUR CODE HERE
import sys
from collections import Counter

def main():
    import sys
    input = sys.stdin.read().split()
    idx = 0
    t = int(input[idx]); idx +=1
    for _ in range(t):
        n = int(input[idx]); k = int(input[idx+1]); idx +=2
        s = input[idx]; idx +=1
        cnt = Counter(s)
        C = sum(1 for v in cnt.values() if v %2 !=0)
        m = n - k
        if m <0:
            print("NO")
            continue
        required = C - (m %2)
        if required <0:
            required =0
        if k >= required and (k - required) %2 ==0:
            print("YES")
        else:
            print("NO")

if __name__ == "__main__":
    main()