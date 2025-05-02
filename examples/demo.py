import warnings
from typing import List

from colorama import Fore, Style, init

from sir.core import SIR
from sir.utils import create_embedding_retriever

# Initialize colorama
init()

# Suppress PEFT warnings
warnings.filterwarnings("ignore", category=UserWarning, module="peft")


def print_hop_results(hop_number: int, query: str, results: list, elapsed_time: float):
    """Print formatted results for a single hop"""
    print(f"\n{Fore.CYAN}{'=' * 80}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}Hop {hop_number}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'=' * 80}{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}Query:{Style.RESET_ALL} {query}")
    print(f"\n{Fore.YELLOW}Retrieved Documents:{Style.RESET_ALL}")
    for i, result in enumerate(results, 1):
        print(
            f"\n{Fore.GREEN}{i}.{Style.RESET_ALL} {Fore.MAGENTA}Similarity: {result.similarity:.4f}{Style.RESET_ALL}"
        )
        print(f"   {Fore.WHITE}Document: {result.document}{Style.RESET_ALL}")
        print(f"   {Fore.BLUE}ID: {result.doc_id}{Style.RESET_ALL}")
    print(f"\n{Fore.YELLOW}Time taken: {elapsed_time:.3f}s{Style.RESET_ALL}")


def reformulate_query(original_query: str, retrieved_docs: List[str]) -> str:
    """Custom reformulation function that focuses on the next piece of information needed"""
    if not retrieved_docs:
        return original_query

    # Extract key entities from retrieved docs
    context = "\n".join([f"{i + 1}. {doc}" for i, doc in enumerate(retrieved_docs)])

    # If we found information about the CTO, focus on finding info about his wife
    if any("CTO" in doc for doc in retrieved_docs):
        return "What is the birthplace of Michael Rodriguez's wife?"

    # If we found information about the wife, focus on finding her birthplace
    if any("wife" in doc.lower() for doc in retrieved_docs):
        return "What is the birthplace of Dr. Julia Rodriguez?"

    # Otherwise, keep original query with context
    return f"{original_query}\n\nContext:\n{context}"


def main():
    documents = [
        "Sarah Chen was born on April 15, 1978, in San Francisco.",
        "Sarah Chen is the CEO of Horizon Technologies.",
        "Horizon Technologies was founded in 2005 by Sarah Chen and Michael Rodriguez.",
        "Michael Rodriguez is the CTO of Horizon Technologies and was born in Austin.",
        "Michael Rodriguez is married to Dr. Julia Rodriguez, a neurologist at San Francisco General Hospital.",
        "Horizon Technologies' headquarters is in San Francisco's SoMa district.",
        "In 2019, Horizon established the Horizon Foundation, headed by Lisa Rodriguez.",
        "Lisa Rodriguez is Michael Rodriguez's sister and obtained her Master's from Harvard.",
        "San Francisco was where Dr. Julia Rodriguez was born and grew up in.",
    ]

    # Example multi-hop query
    query = "What is the birthplace of the Horizon Technologies' CTO's wife?"

    print(f"\n{Fore.CYAN}{'=' * 80}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}Multi-Hop Question Answering Demo{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'=' * 80}{Style.RESET_ALL}")
    print(f"\n{Fore.YELLOW}Query:{Style.RESET_ALL} {query}")
    print(f"\n{Fore.YELLOW}Corpus:{Style.RESET_ALL}")
    for i, doc in enumerate(documents, 1):
        print(f"{Fore.GREEN}{i}.{Style.RESET_ALL} {doc}")

    # Optional: Try with embedding retriever
    try:
        print(f"\n\n{Fore.CYAN}{'=' * 80}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}Starting Multi-Hop Retrieval{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'=' * 80}{Style.RESET_ALL}")

        # Create embedding retriever
        embedding_retriever = create_embedding_retriever(
            documents,
            embedder_config={
                "model_name_or_path": "notebooks/output/sbert-lora-multihop-e5/best_model"
            },
            return_top_k=3,
        )

        # Create SIR with embedding retriever
        sir_with_embeddings = SIR(
            retrieval_fn=embedding_retriever,
            max_hops=3,
            verbose=False,  # We'll handle printing ourselves
        )

        # Run query and get detailed results
        results_by_hop, hop_results = sir_with_embeddings.retrieve(
            query, top_k=3, max_sim=0.8
        )

        # Print results for each hop
        for i, (hop_result, results) in enumerate(zip(hop_results, results_by_hop), 1):
            print_hop_results(i, hop_result.query, results, hop_result.elapsed_time)

        # Get and print final answer
        answer, supporting_docs = sir_with_embeddings.get_answer(query)
        print(f"\n{Fore.CYAN}{'=' * 80}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}Final Answer{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'=' * 80}{Style.RESET_ALL}")
        print(f"\n{Fore.YELLOW}Answer:{Style.RESET_ALL} {answer}")
        print(f"\n{Fore.YELLOW}Supporting Documents:{Style.RESET_ALL}")
        for i, doc in enumerate(supporting_docs, 1):
            print(f"{Fore.GREEN}{i}.{Style.RESET_ALL} {doc}")

    except ImportError:
        print(
            f"\n{Fore.RED}Skipping embedding example - sentence-transformers not installed{Style.RESET_ALL}"
        )


if __name__ == "__main__":
    main()
