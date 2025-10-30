import sys

def main():
    data = sys.stdin.read().split()
    n = int(data[0])
    A = list(map(int, data[1:1+n]))
    q = int(data[1+n])
    queries = []
    index = 1 + n + 1
    for i in range(q):
        L = int(data[index]); R = int(data[index+1]); index += 2
        queries.append((L, R))
    
    out_lines = []
    for (L, R) in queries:
        l0 = L - 1
        r0 = R - 1
        length = r0 - l0 + 1
        mid = l0 + (length + 1) // 2
        i_ptr = l0
        count = 0
        for j in range(mid, r0 + 1):
            if i_ptr >= mid:
                break
            if A[i_ptr] * 2 <= A[j]:
                count += 1
                i_ptr += 1
        out_lines.append(str(count))
    
    sys.stdout.write("
".join(out_lines))

if __name__ == '__main__':
    main()