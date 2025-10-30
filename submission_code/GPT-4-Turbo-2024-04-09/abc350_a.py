def main():
    import sys
    input = sys.stdin.read().strip()
    
    if input[:3] != "ABC":
        print("No")
        return
    
    try:
        number = int(input[3:])
    except ValueError:
        print("No")
        return
    
    if 1 <= number <= 315 and number != 316:
        print("Yes")
    elif 317 <= number <= 349:
        print("Yes")
    else:
        print("No")

if __name__ == "__main__":
    main()