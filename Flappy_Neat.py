import pygame
import neat
import os
import pickle
from main import Bird, Pipe, Base, WIN_WIDTH, WIN_HEIGHT, FLOOR, BG_IMG, draw_window

pygame.font.init()
STAT_FONT = pygame.font.SysFont("comicsans", 30)

GEN = 0

def eval_genomes(genomes, config):
    global GEN
    GEN += 1

    nets = []
    ge = []
    birds = []

    for genome_id, genome in genomes:
        genome.fitness = 0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        nets.append(net)
        ge.append(genome)
        birds.append(Bird(230, 350))

    base = Base(FLOOR)
    pipes = [Pipe(600)]
    score = 0

    win = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
    clock = pygame.time.Clock()

    run = True
    while run and len(birds) > 0:
        clock.tick(30)  # 30 FPS
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                pygame.quit()
                quit()

        pipe_ind = 0
        if len(pipes) > 1 and birds[0].x > pipes[0].x + pipes[0].PIPE_TOP.get_width():
            pipe_ind = 1

        for i, bird in enumerate(birds):
            bird.move()
            ge[i].fitness += 0.1  # reward for staying alive

            # Inputs: bird.y, distance to pipe top, distance to pipe bottom
            output = nets[i].activate((
                bird.y,
                abs(bird.y - pipes[pipe_ind].height),
                abs(bird.y - pipes[pipe_ind].bottom)
            ))

            if output[0] > 0.5:
                bird.jump()

        base.move()

        rem = []
        add_pipe = False
        for pipe in pipes:
            pipe.move()

            for i, bird in enumerate(birds):
                if pipe.collide(bird):
                    ge[i].fitness -= 1
                    birds.pop(i)
                    nets.pop(i)
                    ge.pop(i)
                    continue

            if not pipe.passed and len(birds) > 0 and pipe.x < birds[0].x:
                pipe.passed = True
                add_pipe = True


            if pipe.x + pipe.PIPE_TOP.get_width() < 0:
                rem.append(pipe)

        if add_pipe:
            score += 1
            for genome in ge:
                genome.fitness += 5
            pipes.append(Pipe(600))

        for r in rem:
            pipes.remove(r)

        for i, bird in enumerate(birds):
            if bird.y + bird.img.get_height() >= FLOOR or bird.y < 0:
                birds.pop(i)
                nets.pop(i)
                ge.pop(i)

        draw_training_window(win, birds, pipes, base, score, GEN, len(birds))


def draw_training_window(win, birds, pipes, base, score, gen, alive):
    win.blit(BG_IMG, (0, 0))

    for pipe in pipes:
        pipe.draw(win)

    base.draw(win)
    for bird in birds:
        bird.draw(win)

    # Overlay stats
    score_label = STAT_FONT.render(f"Score: {score}", 1, (255, 255, 255))
    gen_label = STAT_FONT.render(f"Gen: {gen}", 1, (255, 255, 255))
    alive_label = STAT_FONT.render(f"Alive: {alive}", 1, (255, 255, 255))

    win.blit(score_label, (WIN_WIDTH - score_label.get_width() - 10, 10))
    win.blit(gen_label, (10, 10))
    win.blit(alive_label, (10, 50))

    pygame.display.update()


def run(config_path):
    config = neat.config.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path
    )

    population = neat.Population(config)
    population.add_reporter(neat.StdOutReporter(True))
    population.add_reporter(neat.StatisticsReporter())

    winner = population.run(eval_genomes,100)

    # Save best genome
    with open("best_flappy_bird.pkl", "wb") as f:
        pickle.dump(winner, f)


if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_file = os.path.join(local_dir, "Neat_Config.txt")
    run(config_file)
