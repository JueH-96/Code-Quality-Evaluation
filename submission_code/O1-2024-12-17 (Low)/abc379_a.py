def main():
    N = int(input().strip())
    
    # Extract the digits a (hundreds), b (tens), c (ones)
    a = N // 100
    b = (N // 10) % 10
    c = N % 10
    
    # Form the integers by rearranging the digits
    rearranged1 = b * 100 + c * 10 + a
    rearranged2 = c * 100 + a * 10 + b
    
    print(rearranged1, rearranged2)

# Call the main function
main()