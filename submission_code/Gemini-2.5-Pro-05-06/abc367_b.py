# Read the input string
x_str = input()

# Convert the string to a float
# For inputs like "12.340", float() will parse it as 12.34
# For inputs like "1.000", float() will parse it as 1.0
# For inputs like "0.000", float() will parse it as 0.0
num_float = float(x_str)

# Use the :g format specifier.
# This is the "General format".
# For floats, it removes insignificant trailing zeros from the fractional part.
# If the number has no fractional part after removing trailing zeros (i.e., it's an integer),
# it also removes the decimal point.
# Examples:
# "{:g}".format(1.012) -> "1.012"
# "{:g}".format(12.34) -> "12.34" (from 12.340)
# "{:g}".format(99.9)  -> "99.9"  (from 99.900)
# "{:g}".format(0.0)   -> "0"     (from 0.000)
# "{:g}".format(1.0)   -> "1"     (from 1.000)
formatted_num_str = "{:g}".format(num_float)

# Print the formatted number
print(formatted_num_str)