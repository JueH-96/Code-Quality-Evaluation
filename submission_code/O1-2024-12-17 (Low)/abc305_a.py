def main():
    import sys
    
    N = int(sys.stdin.readline().strip())
    # Compute which multiple of 5 is closest to N
    nearest_station = 5 * round(N / 5)
    print(nearest_station)

# Call the main function
main()