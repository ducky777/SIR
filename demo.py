import argparse

from sir.core import SIR, DocStorage


def main():
    """Example usage of the SIR pipeline."""
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Run SIR pipeline with a custom question"
    )
    parser.add_argument(
        "question",
        type=str,
        default="Which companies own multiple social media platforms?",
        help="The question to ask the SIR pipeline",
    )
    args = parser.parse_args()

    from sentence_transformers import SentenceTransformer

    # Load model
    model_name = "duckduckpuck/sir-sbert-e5-large-v1"  # Change as needed
    model = SentenceTransformer(model_name)

    # Example sentences
    sentences = [
        # Step 1: Platform Definitions (mention 'social media')
        "Glimmer is a mobile-first social media platform focused on voice-based status updates.",
        "Chattr is a video reply platform designed for asynchronous group conversations, often used like a social media network.",
        "Streamlet is an animated meme-sharing app categorized under niche social platforms.",
        "Flickly allows users to post minimalist photo stories and is widely seen as a photo-based social media tool.",
        "EchoBay started as a podcast aggregator but evolved into an audio-first social platform with community commenting.",
        # Step 2: Platform Ownership (no mention of 'social media')
        "Alliance Technologies acquired Glimmer in a private deal in 2021 to expand its communication tools division.",
        "EchoBay is currently owned by Bluenova Holdings, which focuses on digital content aggregation.",
        "Chattr became a subsidiary of Bluenova Holdings following its merger with the audio startup Audina.",
        "Borealis Labs invested heavily in Flickly and owns 60% of the company as part of its incubator program.",
        "Streamlet was acquired by DeltaSync as part of its broader push into user-generated content platforms.",
        # Step 3: Company Descriptions
        "Alliance Technologies is a global holding company that owns several communication and productivity startups.",
        "Bluenova Holdings is a media conglomerate focused on entertainment, audio, and social ecosystems.",
        "DeltaSync is a tech group with a focus on immersive content platforms, including AR and short-form media.",
        "Borealis Labs is a research-focused incubator that spins out consumer-facing apps through venture partnerships.",
        # Irrelevant but related (clutter)
        "Glimmer was founded in 2020 by three former university classmates who initially aimed to build a voice diary app.",
        "Chattr received seed funding from Coastal Ventures and was initially incubated at a university startup hub.",
        "Streamlet ran a controversial campaign in 2022 involving meme-based political ads that drew regulatory scrutiny.",
        "Flickly recently introduced a dark mode and auto-captioning feature to improve accessibility.",
        "EchoBay launched a premium tier that includes exclusive podcast bundles and community-hosted live rooms.",
        "Alliance Technologies is also rumored to be exploring quantum communication protocols for enterprise security.",
        "Bluenova Holdings attempted to acquire a news curation platform in 2022, but the deal fell through.",
        "DeltaSync's founder was recently interviewed in Wired about the future of interactive video memes.",
        "Borealis Labs recently received a government grant to explore AI safety alignment in content recommendation.",
        "Alliance's annual developer conference, ComNet, focuses on next-gen voice technologies.",
        "Chattr has struggled with moderation in emerging markets due to language model inaccuracies.",
        "DeltaSync briefly explored wearable AR glasses before shutting the unit down in 2023.",
        "Bluenova's CEO is known for her aggressive acquisition strategy across Southeast Asia.",
        "EchoBay users in Germany reported outages during a major product update in late 2023.",
        "Flickly's name is derived from the Icelandic word for 'snapshot', though the company is based in Toronto.",
        "Borealis Labs has a podcast series featuring interviews with its incubated founders and CTOs.",
    ]

    # Create document storage and add documents
    docs = DocStorage(model)
    docs.add_docs(sentences)

    # Create SIR pipeline with cross-encoder enabled
    sir_pipeline = SIR(model=model, docs=docs, use_cross_encoder=False)

    # Run query with hybrid search using the provided question
    _ = sir_pipeline.run(
        query=args.question, top_k=1, max_hops=10, use_hybrid=True, bm25_weight=0.3
    )


if __name__ == "__main__":
    main()

"""
Expected Output:
┏━━━━━━━┳━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┓
┃ Hop   ┃ #   ┃ Document                                                                                                           ┃ Score      ┃
┡━━━━━━━╇━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━┩
│ 1     │ 1   │ Glimmer is a mobile-first social media platform focused on voice-based status updates.                             │ 0.3181     │ <--- Identified a social media platform
│       │     │ ------------------------------                                                                                     │            │
│ 2     │ 1   │ Bluenova Holdings is a media conglomerate focused on entertainment, audio, and social ecosystems.                  │ 0.4684     │ <--- Found a parent company with multiple holdings
│ 3     │ 1   │ EchoBay is currently owned by Bluenova Holdings, which focuses on digital content aggregation.                     │ 0.4406     │ <--- Mapped a platform (EchoBay) to its parent (Bluenova)
│ 4     │ 1   │ EchoBay started as a podcast aggregator but evolved into an audio-first social platform with community commenting. │ 0.4435     │ <--- Identified EchoBay as a social platform
│ 5     │ 1   │ Bluenova Holdings attempted to acquire a news curation platform in 2022, but the deal fell through.                │ 0.4449     │ <--- Distractor: acquisition attempt unrelated to current platforms
│ 6     │ 1   │ DeltaSync is a tech group with a focus on immersive content platforms, including AR and short-form media.          │ 0.4252     │ <--- Found another potential parent company (DeltaSync)
│ 7     │ 1   │ Flickly allows users to post minimalist photo stories and is widely seen as a photo-based social media tool.       │ 0.4322     │ <--- Identified another social media platform (Flickly)
│ 8     │ 1   │ Borealis Labs invested heavily in Flickly and owns 60% of the company as part of its incubator program.            │ 0.4500     │ <--- Mapped Flickly to its partial owner (Borealis Labs)
│ 9     │ 1   │ Streamlet was acquired by DeltaSync as part of its broader push into user-generated content platforms.             │ 0.5010     │ <--- Mapped Streamlet to its parent (DeltaSync)
│ 10    │ 1   │ Streamlet is an animated meme-sharing app categorized under niche social platforms.                                │ 0.4700     │ <--- Identified Streamlet as a social media platform
└───────┴─────┴────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┴────────────┘
"""
