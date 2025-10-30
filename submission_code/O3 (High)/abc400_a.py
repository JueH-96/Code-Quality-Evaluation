import sys

def main():
    A_input = sys.stdin.readline().strip()
    if not A_input:
        return
    A = int(A_input)
    total_people = 400
    
    if total_people % A == 0:
        print(total_people // A)
    else:
        print(-1)

if __name__ == "__main__":
    main()