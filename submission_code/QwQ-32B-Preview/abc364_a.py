import sys

def main():
    data = sys.stdin.read().splitlines()
    N = int(data[0])
    S = data[1:N+1]
    
    for i in range(N-1):
        if S[i] == "sweet" and S[i+1] == "sweet":
            if i+1 < N-1:
                print("No")
                return
            else:
                print("Yes")
                return
    print("Yes")

if __name__ == "__main__":
    main()