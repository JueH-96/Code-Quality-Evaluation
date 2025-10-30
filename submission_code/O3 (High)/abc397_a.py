from decimal import Decimal
import sys

def main() -> None:
    # Read the temperature as a string and convert to Decimal for exact comparison
    X = Decimal(sys.stdin.readline().strip())

    if X >= Decimal('38.0'):
        print(1)  # High fever
    elif X >= Decimal('37.5'):
        print(2)  # Fever
    else:
        print(3)  # Normal

if __name__ == "__main__":
    main()