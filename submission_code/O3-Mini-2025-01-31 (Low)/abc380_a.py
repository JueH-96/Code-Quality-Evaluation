def main():
    import sys
    input_data = sys.stdin.read().strip()
    if not input_data:
        return
    N = input_data.strip()
    # Count occurrences of digits.
    count1 = N.count('1')
    count2 = N.count('2')
    count3 = N.count('3')
    
    if count1 == 1 and count2 == 2 and count3 == 3:
        print("Yes")
    else:
        print("No")

if __name__ == '__main__':
    main()