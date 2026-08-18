from predictor import predict_news


print("=" * 60)
print("              FAKE NEWS DETECTOR")
print("=" * 60)

while True:
    article = input("\nEnter your news article (or type 'exit' to quit):\n")

    if article.lower() == "exit":
        print("\nThank you for using Fake News Detector!")
        break

    if not article.strip():
        print("Please enter some news text.")
        continue

    result = predict_news(article)

    if result == "FAKE NEWS":
        print("\n🔴 Prediction: FAKE NEWS")
    else:
        print("\n🟢 Prediction: REAL NEWS")