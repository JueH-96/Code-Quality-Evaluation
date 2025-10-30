import sys
input = sys.stdin.read

def main():
    data = input().split()
    N = int(data[0])
    K = int(data[1])
    S = data[2]
    
    count = 0
    i = 0
    while i <= N - K:
        if S[i:i+K] == 'O' * K:
            count += 1
            i += K  # Move past this segment to avoid overlapping
        else:
            i += 1
    
    print(count)

if __name__ == "__main__":
    main()