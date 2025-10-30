import sys
input = sys.stdin.read

def main():
    data = input().split()
    index = 0
    
    N = int(data[index])
    index += 1
    M = int(data[index])
    index += 1
    
    goals = [int(data[index + i]) for i in range(M)]
    index += M
    
    nutrients = [0] * M
    
    for _ in range(N):
        for j in range(M):
            nutrients[j] += int(data[index])
            index += 1
    
    for i in range(M):
        if nutrients[i] < goals[i]:
            print("No")
            return
    
    print("Yes")

if __name__ == "__main__":
    main()