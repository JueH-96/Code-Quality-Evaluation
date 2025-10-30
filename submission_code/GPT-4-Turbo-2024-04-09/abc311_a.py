import sys
input = sys.stdin.read

def main():
    data = input().split()
    N = int(data[0])
    S = data[1]
    
    seen = set()
    for i in range(N):
        seen.add(S[i])
        if len(seen) == 3:
            print(i + 1)
            break

if __name__ == "__main__":
    main()