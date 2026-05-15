"""Query classification for domain boundary enforcement and intent routing."""
import re
from enum import Enum
from typing import NamedTuple


class QueryIntent(Enum):
    """Query intent classification."""
    ASSESSMENT_RECOMMENDATION = "assessment_recommendation"
    ASSESSMENT_COMPARISON = "assessment_comparison"
    OFF_TOPIC = "off_topic"
    PROMPT_INJECTION = "prompt_injection"


class ClassificationResult(NamedTuple):
    """Classification result with intent and metadata."""
    intent: QueryIntent
    confidence: float
    reason: str
    comparison_items: list[str] = []


def _tokenize(text: str) -> list[str]:
    """Simple tokenization for classification."""
    return re.findall(r"[a-z0-9\-]+", (text or "").lower())


def classify_query(query: str) -> ClassificationResult:
    """
    Classify a query into one of the predefined intents.
    
    Returns a ClassificationResult with intent, confidence, reason, and optional metadata.
    """
    query_lower = query.lower()
    
    # Check for prompt injection patterns first (highest priority refusal).
    injection_phrases = [
        "ignore previous",
        "ignore all previous",
        "ignore instructions",
        "do not follow earlier",
        "forget everything",
        "override",
        "bypass",
        "don't follow",
    ]
    if any(p in query_lower for p in injection_phrases):
        return ClassificationResult(
            intent=QueryIntent.PROMPT_INJECTION,
            confidence=0.95,
            reason="Detected instruction override attempt",
        )
    
    # Check for off-topic patterns.
    off_topic_keywords = {
        "legal": ["legal", "law", "attorney", "lawsuit", "contract", "compliance"],
        "medical": ["medical", "diagnosis", "doctor", "prescription", "health"],
        "hr_management": ["salary", "compensation", "bonus", "raise", "firing", "fire", "terminate", "termination", "hiring process", "recruitment strategy"],
    }
    
    for category, keywords in off_topic_keywords.items():
        if any(kw in query_lower for kw in keywords):
            return ClassificationResult(
                intent=QueryIntent.OFF_TOPIC,
                confidence=0.90,
                reason=f"Off-topic: {category} advice is not within scope",
            )
    
    # Check for comparison queries (e.g., "Difference between OPQ and GSA?").
    comparison_triggers = [
        "difference between",
        "compare",
        "vs",
        "versus",
        "which is better",
        "compare these",
        "comparison",
    ]
    
    if any(trigger in query_lower for trigger in comparison_triggers):
        # Extract assessment names mentioned.
        assessment_names = re.findall(r"([A-Z][A-Z0-9]*)", query)
        if assessment_names:
            return ClassificationResult(
                intent=QueryIntent.ASSESSMENT_COMPARISON,
                confidence=0.85,
                reason="Comparison request detected",
                comparison_items=assessment_names,
            )
    
    # Check for assessment-related queries.
    assessment_keywords = [
        "assessment",
        "evaluate",
        "test",
        "evaluate",
        "candidate",
        "hire",
        "recruit",
        "skill",
        "competency",
        "personality",
        "aptitude",
        "cognitive",
        "ability",
    ]
    
    # Look for tech roles or skills as well.
    tech_keywords = [
        "python",
        "java",
        "javascript",
        "backend",
        "frontend",
        "fullstack",
        "data engineer",
        "devops",
        "cloud",
        "qa",
        "developer",
        "engineer",
    ]
    
    assessment_score = sum(1 for kw in assessment_keywords if kw in query_lower)
    tech_score = sum(1 for kw in tech_keywords if kw in query_lower)
    total_score = assessment_score + tech_score
    
    if total_score >= 1:
        confidence = min(0.5 + (total_score * 0.15), 0.95)
        reason = f"Assessment-related (score: {total_score})"
        # Treat VERY generic "assessment" queries as vague/low-confidence (e.g., just "I need an assessment")
        # But don't flag specific requests like "personality tests" or "Also add java test" as vague
        if (
            assessment_score >= 1
            and tech_score == 0
            and len(query.strip()) < 12
            and all(kw in ["assessment", "test", "evaluate", "evaluates"] for kw in _tokenize(query))
        ):
            confidence = min(confidence * 0.6, 0.5)
            reason += "; vague generic request"
        # Also flag queries with ONLY "assessment" and no descriptors as vague
        # (e.g., "I need an assessment" has only assessment keyword, no personality, java, etc.)
        elif (
            assessment_score >= 1
            and tech_score == 0
            and "assessment" in query_lower
            and not any(
                kw in query_lower
                for kw in ["personality", "aptitude", "cognitive", "ability", "java", "python"]
            )
        ):
            # If it has only generic assessment keywords without specificity
            if len(query.strip()) < 25:
                confidence = 0.4  # Strongly mark as vague/needs clarification
                reason = "Vague assessment request; needs role/type clarification"
        
        return ClassificationResult(
            intent=QueryIntent.ASSESSMENT_RECOMMENDATION,
            confidence=confidence,
            reason=reason,
        )
    
    # Default: if query is minimal, ask for clarification instead of classifying as off-topic.
    if len(query.strip()) < 8:
        return ClassificationResult(
            intent=QueryIntent.ASSESSMENT_RECOMMENDATION,
            confidence=0.3,
            reason="Vague query; clarification needed",
        )
    
    # Fallback: uncertain but assume assessment-related.
    return ClassificationResult(
        intent=QueryIntent.ASSESSMENT_RECOMMENDATION,
        confidence=0.2,
        reason="Uncertain intent; defaulting to assessment mode",
    )
