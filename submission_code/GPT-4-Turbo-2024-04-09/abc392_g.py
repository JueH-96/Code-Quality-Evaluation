import sys
input = sys.stdin.read

def main():
    data = input().split()
    N = int(data[0])
    S = list(map(int, data[1:]))
    
    S.sort()
    count = 0
    S_set = set(S)
    
    for i in range(N):
        for j in range(i + 1, N):
            A = S[i]
            B = S[j]
            C = 2 * B - A
            if C in S_set and C > B:
                count += 1
    
    print(count)

main()