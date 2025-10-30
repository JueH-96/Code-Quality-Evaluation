import sys

# Helper function to check if a number is a repdigit
# and return its repeating character if it is.
# Otherwise, returns None.
def is_repdigit_and_get_char(num_val):
    s = str(num_val)
    # num_val is at least 1, so s is never empty and s[0] is valid.
    first_char = s[0]
    
    # Check if all characters in s are the same as first_char.
    # A concise way is to create a string of first_char repeated len(s) times
    # and compare it with s.
    # e.g., for num_val=22, s="22", first_char='2'. '2'*len(s) = "22". Matches.
    # e.g., for num_val=12, s="12", first_char='1'. '1'*len(s) = "11". No match.
    if s == first_char * len(s):
        return first_char
    else:
        return None

def main():
    N = int(sys.stdin.readline())
    
    # Read the list of D_i values (number of days in month i)
    D_str_list = sys.stdin.readline().split()
    # Convert to list of integers. D_list is 0-indexed.
    # D_list[k] corresponds to D_{k+1} from problem statement.
    D_list = [int(x) for x in D_str_list]

    count = 0 # Initialize total count of repdigit dates

    # Iterate through each month, from 1 to N (1-indexed)
    for i_month_1_indexed in range(1, N + 1):
        # Check if the current month number is a repdigit
        rep_char_month = is_repdigit_and_get_char(i_month_1_indexed)
        
        if rep_char_month is not None:
            # If month is a repdigit, rep_char_month is its repeating digit (as a character)
            
            # Candidate day 1: Day number is the single digit itself.
            # e.g., if rep_char_month is '1', day_val1 becomes 1.
            day_val1 = int(rep_char_month)
            
            # Number of days in current month i_month_1_indexed is D_list[i_month_1_indexed - 1]
            if day_val1 <= D_list[i_month_1_indexed - 1]:
                count += 1
                
            # Candidate day 2: Day number is the two-digit repdigit.
            # e.g., if rep_char_month is '1', string "11" is formed, day_val2 becomes 11.
            # rep_char_month * 2 performs string repetition, e.g., '1'*2 = "11".
            day_val2 = int(rep_char_month * 2) 
            
            # day_val2 will be one of 11, 22, ..., 99. All are <= 100.
            # Check if this day is valid for the current month.
            if day_val2 <= D_list[i_month_1_indexed - 1]:
                count += 1
                
    # Print the final count
    sys.stdout.write(str(count) + "
")

if __name__ == '__main__':
    main()