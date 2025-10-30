def main():
    import sys
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    n = int(input_data[0])
    # Dictionary to store the minimum deliciousness for each color
    color_min = {}
    pos = 1
    for _ in range(n):
        deliciousness = int(input_data[pos])
        color = int(input_data[pos+1])
        pos += 2
        
        # Update the minimum for that color
        if color in color_min:
            color_min[color] = min(color_min[color], deliciousness)
        else:
            color_min[color] = deliciousness
            
    # Among all colors, we want to select the color with the maximum (of minimum deliciousness)
    answer = max(color_min.values())
    print(answer)

# Call main to run the solution
if __name__ == '__main__':
    main()