def main():
    import sys
    
    N = sys.stdin.readline().strip()
    
    # Count occurrences of '1', '2', and '3'
    count1 = N.count('1')
    count2 = N.count('2')
    count3 = N.count('3')
    
    # Check the conditions
    if count1 == 1 and count2 == 2 and count3 == 3:
        print("Yes")
    else:
        print("No")

if __name__ == "__main__":
    main()