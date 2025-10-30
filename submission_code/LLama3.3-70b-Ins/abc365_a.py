def find_days_in_year(year):
    """
    This function calculates the number of days in a given year of the Gregorian calendar.

    Args:
        year (int): The year for which the number of days needs to be calculated.

    Returns:
        int: The number of days in the given year.
    """
    # If the year is not a multiple of 4, it has 365 days
    if year % 4 != 0:
        return 365
    
    # If the year is a multiple of 4 but not a multiple of 100, it has 366 days
    elif year % 100 != 0:
        return 366
    
    # If the year is a multiple of 100 but not a multiple of 400, it has 365 days
    elif year % 400 != 0:
        return 365
    
    # If the year is a multiple of 400, it has 366 days
    else:
        return 366


# Read the input from stdin
year = int(input())

# Calculate and print the number of days in the given year
print(find_days_in_year(year))