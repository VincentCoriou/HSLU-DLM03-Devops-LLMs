def calculate_area(length, width):
    verbose = True
    result = length * width

    if length > 0 and width > 0:
        print("The area is:", result)
    return result

if __name__ == "__main__":
    A = calculate_area(2, 5)
    print(A)
