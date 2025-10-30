import sys
input = sys.stdin.read

def main():
    data = input().split()
    N = int(data[0])
    dishes = data[1:]
    
    for i in range(1, N):
        if dishes[i] == "sweet" and dishes[i-1] == "sweet":
            print("No")
            return
    
    print("Yes")

if __name__ == "__main__":
    main()