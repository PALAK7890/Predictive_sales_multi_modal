from ollama import chat


class CampaignConsultant:

    def __init__(
        self,
        model="llama3.2",
    ):
        self.model = model

    def ask(
        self,
        prompt: str,
    ):

        response = chat(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
        )

        return response["message"]["content"]