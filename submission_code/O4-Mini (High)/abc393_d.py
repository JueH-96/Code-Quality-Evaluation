import sys

def main():
    data = sys.stdin.readline().split()
    if not data:
        return
    n = int(data[0])
    S = sys.stdin.readline().strip()
    
    # Collect positions of '1's
    pos = []
    for i, ch in enumerate(S):
        if ch == '1':
            pos.append(i)
    
    k = len(pos)
    # a[i] = pos[i] - i; this sequence is non-decreasing
    a = [pos[i] - i for i in range(k)]
    
    # Best starting index l is median of a[]
    median = a[k // 2]
    
    # Sum of moves = sum |a[i] - median|
    ans = 0
    for x in a:
        ans += abs(x - median)
    
    print(ans)

if __name__ == "__main__":
    main()