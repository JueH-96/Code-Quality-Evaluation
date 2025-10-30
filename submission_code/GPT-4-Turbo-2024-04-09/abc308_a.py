def main():
    import sys
    input = sys.stdin.read
    data = list(map(int, input().split()))
    
    # Check if the sequence is monotonically non-decreasing
    if not all(data[i] <= data[i+1] for i in range(len(data)-1)):
        print("No")
        return
    
    # Check if all numbers are between 100 and 675, inclusive
    if not all(100 <= x <= 675 for x in data):
        print("No")
        return
    
    # Check if all numbers are multiples of 25
    if not all(x % 25 == 0 for x in data):
        print("No")
        return
    
    # If all conditions are satisfied
    print("Yes")

if __name__ == "__main__":
    main()