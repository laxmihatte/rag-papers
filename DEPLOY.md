# Deploy free on Hugging Face Spaces

This puts the app online for free, always-on, until you delete the Space.
No server, no VM, no credit card.

## 1. Get a free Groq API key (for generating answers)

1. Go to <https://console.groq.com> and sign in.
2. Create an API key and copy it. (You'll paste it into the Space in step 3.)

## 2. Create the Space and push this repo to it

1. Go to <https://huggingface.co/new-space>.
2. Name it `rag-papers`, choose **SDK: Gradio**, hardware **CPU basic (free)**.
3. It gives you a git URL like `https://huggingface.co/spaces/<user>/rag-papers`.
4. Push this repo to it as a second remote:

   ```bash
   git remote add space https://huggingface.co/spaces/<user>/rag-papers
   git push space main
   ```

   (You'll be asked for your Hugging Face username and an access token as the
   password — create one at <https://huggingface.co/settings/tokens>, role **Write**.)

## 3. Add your Groq key as a Space secret

In the Space: **Settings → Variables and secrets → New secret**

- Name: `GROQ_API_KEY`
- Value: the key from step 1

The Space restarts, builds the index on first boot (~1 min), and goes live at:

```
https://<user>-rag-papers.hf.space
```

## Updating later

Push to both remotes:

```bash
git push origin main    # GitHub (source of truth)
git push space main     # redeploys the Space
```

## Taking it down

Delete the Space in its **Settings**. That's it — nothing else is running anywhere.
```
