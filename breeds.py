import json

# List of dog breeds
breeds = [
    'Affenpinscher', 'Afghan Hound', 'African Hunting Dog', 'Airedale',
    'American Staffordshire Terrier', 'Appenzeller',
    'Australian Terrier', 'Basenji', 'Basset', 'Beagle',
    'Bedlington Terrier', 'Bernese Mountain Dog',
    'Black-and-Tan Coonhound', 'Blenheim Spaniel', 'Bloodhound',
    'Bluetick', 'Border Collie', 'Border Terrier', 'Borzoi',
    'Boston Bull', 'Bouvier des Flandres', 'Boxer',
    'Brabancon Griffon', 'Briard', 'Brittany Spaniel', 'Bull Mastiff',
    'Cairn', 'Cardigan', 'Chesapeake Bay Retriever', 'Chihuahua',
    'Chow', 'Clumber', 'Cocker Spaniel', 'Collie',
    'Curly-Coated Retriever', 'Dandie Dinmont', 'Dhole', 'Dingo',
    'Doberman', 'English Foxhound', 'English Setter',
    'English Springer', 'Entlebucher', 'Eskimo Dog',
    'Flat-Coated Retriever', 'French Bulldog', 'German Shepherd',
    'German Short-Haired Pointer', 'Giant Schnauzer',
    'Golden Retriever', 'Gordon Setter', 'Great Dane',
    'Great Pyrenees', 'Greater Swiss Mountain Dog', 'Groenendael',
    'Ibizan Hound', 'Irish Setter', 'Irish Terrier',
    'Irish Water Spaniel', 'Irish Wolfhound', 'Italian Greyhound',
    'Japanese Spaniel', 'Keeshond', 'Kelpie', 'Kerry Blue Terrier',
    'Komondor', 'Kuvasz', 'Labrador Retriever', 'Lakeland Terrier',
    'Leonberg', 'Lhasa', 'Malamute', 'Malinois', 'Maltese Dog',
    'Mexican Hairless', 'Miniature Pinscher', 'Miniature Poodle',
    'Miniature Schnauzer', 'Newfoundland', 'Norfolk Terrier',
    'Norwegian Elkhound', 'Norwich Terrier', 'Old English Sheepdog',
    'Otterhound', 'Papillon', 'Pekinese', 'Pembroke', 'Pomeranian',
    'Pug', 'Redbone', 'Rhodesian Ridgeback', 'Rottweiler',
    'Saint Bernard', 'Saluki', 'Samoyed', 'Schipperke',
    'Scotch Terrier', 'Scottish Deerhound', 'Sealyham Terrier',
    'Shetland Sheepdog', 'Shih-Tzu', 'Siberian Husky', 'Silky Terrier',
    'Soft-Coated Wheaten Terrier', 'Staffordshire Bullterrier',
    'Standard Poodle', 'Standard Schnauzer', 'Sussex Spaniel',
    'Tibetan Mastiff', 'Tibetan Terrier', 'Toy Poodle', 'Toy Terrier',
    'Vizsla', 'Walker Hound', 'Weimaraner', 'Welsh Springer Spaniel',
    'West Highland White Terrier', 'Whippet',
    'Wire-Haired Fox Terrier', 'Yorkshire Terrier'
]

# Create a dictionary mapping indices to breed names
breed_labels = {str(i): breed for i, breed in enumerate(breeds)}

# Save as a JSON file
with open('Data/breed_labels.json', 'w') as f:
    json.dump(breed_labels, f, indent=4)

print("Breed labels saved to 'breed_labels.json'")
