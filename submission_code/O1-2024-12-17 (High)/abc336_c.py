def main():
    import sys
    N = int(sys.stdin.readline().strip())
    
    # Special case: the 1st good integer is 0
    if N == 1:
        print(0)
        return
    
    # Convert (N-1) to base 5
    n = N - 1
    digits_in_base5 = []
    while n > 0:
        digits_in_base5.append(n % 5)
        n //= 5
    
    # Reverse to get the correct order
    digits_in_base5.reverse()
    
    # Map each base-5 digit d to 2*d (thus digits 0->0,1->2,2->4,3->6,4->8)
    mapped_digits = [str(d * 2) for d in digits_in_base5]
    
    # Join and output
    print("".join(mapped_digits))

# Do NOT forget to call main()    
if __name__ == "__main__":
    main()