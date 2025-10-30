import sys

def main():
    import sys
    import math
    input = sys.stdin.read
    data = input().split()
    idx = 0
    t = int(data[idx])
    idx +=1
    results = []
    for _ in range(t):
        n = int(data[idx])
        k = int(data[idx+1])
        s = data[idx+2]
        idx +=3
        B = [i for i, c in enumerate(s) if c == 'B']
        if not B:
            results.append(0)
            continue
        count =0
        i =0
        m = len(B)
        while i < m:
            start = B[i]
            end = start + k -1
            # Find the farthest j where B[j] <= end
            j = i
            while j < m and B[j] <= end:
                j +=1
            count +=1
            i = j
        results.append(count)
    print('
'.join(map(str, results)))

if __name__ == "__main__":
    main()