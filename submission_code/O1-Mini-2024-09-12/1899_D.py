# YOUR CODE HERE
import sys
from collections import defaultdict

def main():
    import sys
    import sys
    input = sys.stdin.read
    data = input().split()
    idx = 0
    t = int(data[idx]); idx +=1
    for _ in range(t):
        n = int(data[idx]); idx +=1
        a = data[idx:idx+n]; idx +=n
        freq = defaultdict(int)
        for num in a:
            freq[num] +=1
        total =0
        for cnt in freq.values():
            if cnt >=2:
                total += cnt * (cnt -1) //2
        if '1' in freq and '2' in freq:
            total += freq['1'] * freq['2']
        print(total)

if __name__ == "__main__":
    main()