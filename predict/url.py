from model.predict import pred


def predict_url(text):
    flair = pred(text)
    return flair
