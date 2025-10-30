import sys

def main() -> None:
    data = sys.stdin.read().strip().split()
    if not data:
        return
    n = int(data[0])
    numbers = list(map(int, data[1:1 + n]))
    
    # Check if all elements are equal by comparing the size of the set
    if len(set(numbers)) == 1:
        print("Yes")
    else:
        print("No")

if __name__ == "__main__":
    main()