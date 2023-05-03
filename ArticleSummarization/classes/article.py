class Article:
    
    def __init__(self, filename, content):
        self.filename = filename
        self.content = content

    def get_filename(self):
        return self.filename
    
    def get_content(self):
        return self.content