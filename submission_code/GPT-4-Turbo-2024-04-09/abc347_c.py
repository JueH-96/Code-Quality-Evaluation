import sys
input = sys.stdin.read

def main():
    data = input().split()
    N = int(data[0])
    A = int(data[1])
    B = int(data[2])
    D = list(map(int, data[3:]))
    
    week_length = A + B
    
    for d in D:
        day_of_week = (d - 1) % week_length + 1
        if not (1 <= day_of_week <= A):
            print("No")
            return
    
    print("Yes")

if __name__ == "__main__":
    main()