def main():
    import sys
    input_str = sys.stdin.read().strip()
    
    # Find positions of the first and last '|' character
    first_pipe = input_str.find('|')
    last_pipe = input_str.rfind('|')
    
    # Construct the resulting string by excluding the segment between the pipes (inclusive)
    result = input_str[:first_pipe] + input_str[last_pipe+1:]
    
    print(result)

if __name__ == '__main__':
    main()