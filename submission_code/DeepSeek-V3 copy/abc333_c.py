def main():
    N = int(input())
    repunits = []
    # Generate repunits up to a certain length to cover the possible sums
    # Since the maximum N is 333, we need to generate enough repunits
    # The sum of three repunits can be up to 3 * 111111111111 (12 digits)
    # So we generate repunits up to 12 digits
    for i in range(1, 13):
        repunits.append(int('1' * i))
    
    # To store all possible sums of three repunits
    sums = set()
    for a in repunits:
        for b in repunits:
            for c in repunits:
                sums.add(a + b + c)
    
    # Convert the set to a sorted list
    sorted_sums = sorted(sums)
    
    # Print the N-th smallest sum
    print(sorted_sums[N-1])

if __name__ == "__main__":
    main()