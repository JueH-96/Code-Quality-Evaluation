def main():
    N = input().strip()
    digits = [int(d) for d in N]
    
    # Check if every digit is strictly greater than the next
    for i in range(len(digits) - 1):
        if digits[i] <= digits[i + 1]:
            print("No")
            return
    
    print("Yes")

main()