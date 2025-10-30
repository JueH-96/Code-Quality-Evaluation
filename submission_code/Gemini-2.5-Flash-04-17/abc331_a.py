# Read M and D
M, D = map(int, input().split())

# Read y, m, d
y, m, d = map(int, input().split())

# Calculate the next day

# Start with potential next date (incrementing day)
y_next = y
m_next = m
d_next = d + 1

# Check for day overflow
if d_next > D:
    d_next = 1 # Day overflows, reset to 1
    m_next = m + 1 # Attempt to increment month

    # Check for month overflow
    if m_next > M:
        m_next = 1 # Month overflows, reset to 1
        y_next = y + 1 # Year increments
    # else: month increments, year stays the same (m_next = m + 1, y_next = y)
# else: day does not overflow, month and year stay the same (d_next = d + 1, m_next = m, y_next = y)


# Print the result
print(y_next, m_next, d_next)