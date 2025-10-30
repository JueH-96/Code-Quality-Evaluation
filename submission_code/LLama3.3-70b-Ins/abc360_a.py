def solve():
    # Read the input string
    s = input()

    # Find the indices of 'R' and 'M'
    rice_index = s.find('R')
    miso_index = s.find('M')

    # Check if the plate of rice is to the left of the plate of miso soup
    if rice_index < miso_index:
        print("Yes")
    else:
        print("No")

# Call the function
solve()